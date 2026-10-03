# Whole-interval tearing-mode labels and magnetic-rule recovery

The target is a **strong rotating n=1/n=2 magnetic mode**, a tearing-mode proxy without independent island identification. `tm-ours` measures recovery of this magnetic rule; these results do not establish a better TM detector. AUROC and AUPRC are the primary comparisons; F1 depends on calibration.

All new labeling, feature screening, fetching, agreement, fitting, scoring, coverage and galleries exclude the 50 cohort blind shots before opening their signals. The current cohort table contains the 450 development shots. Population results also exclude those IDs. Previous blind result JSONs were moved without inspection into `results/quarantine_blind_test/`.

## Rule and uncertainty

Both raw RMS and its 5 ms median must exceed 12 G (n1) or 6 G (n2) continuously for 50 ms, before joining runs. The n1 convention follows Farre-Kaga; n2 is a local extension. Each qualified seed extends to max(1 G, 10% of its peak). Available release gaps up to 50 ms may join; acquisition gaps cannot. The frozen development harmonic veto is n2/n1 ≤0.57. EFIT rational surfaces alone do not determine m; no ECE island radius has been resolved, and m remains unassigned.

Seed and span frequency evidence now uses `coherent_frequency`, not simply N1FREQ/N2FREQ lying in 1–30 kHz. It tests a local 50 ms percentile-stable frequency window with p90−p10 width ≤max(2 kHz, 25% of its median), and ≥80% actual coherent support over a span. Mirnov fallback requires n-resolved fit ≥0.9, prominence ≥10 dB, 1–30 kHz, and coherent amplitude above the frozen development quiet p95. Available >30 kHz records veto that mode's fallback. These criteria are magnetic proxies, not proof of an island.

Weak tracks require a 100 ms coherent core and are uncertain, extended at the weak-line release floor across ≤50 ms evidence gaps. Population shots run the same Mirnov weak screening wherever existing raw inputs allow. Unscreened time above the frozen weak RMS thresholds (n1 2.0282 G, n2 1.8280 G) is uncertain rather than absent. Missing inputs are disclosed in metadata. Quiet time can be absent even without a weak-line screen. Weak modes that fail the screen still remain cohort **absent**: the weak 7 kHz n=2 line on 189879 is uncertain until 3.9 s; its fading continuation, below the fit/prominence screen, is absent. A review of the earlier labels saw the line to about 4.5 s. This is a strong-mode label, not exhaustive TM truth.

### Criterion pass rates

The source records below report the old in-range test, true coherent frequency and Mirnov fit/prominence criteria separately on absent and present samples, on the catalog-window RMS grid. Missing criterion inputs do not count as failures: the source gives their support denominators. These are diagnostic associations with the magnetic rule, not independent validation.

| Set / n | Criterion | Absent pass % (input s) | Present pass % (input s) |
|---|---|---:|---:|
| cohort / 1 | rms_above_seed | 0.01 (1282.13) | 32.81 (142.29) |
| cohort / 1 | median5ms_above_seed | 0.00 (1282.13) | 33.00 (142.29) |
| cohort / 1 | rms_above_weak | 1.16 (1282.13) | 73.30 (142.29) |
| cohort / 1 | frequency_in_range_legacy | 78.67 (400.54) | 98.36 (142.29) |
| cohort / 1 | frequency_coherent50ms | 0.58 (400.54) | 60.33 (142.29) |
| cohort / 1 | mirnov_phase_fit | 12.87 (1279.07) | 68.34 (142.29) |
| cohort / 1 | mirnov_prominence | 71.86 (1279.07) | 99.51 (142.29) |
| cohort / 1 | span_coherent_support | 4.75 (1282.13) | 70.20 (142.29) |
| cohort / 1 | seed_coherent_support | 1.19 (1282.13) | 60.33 (142.29) |
| cohort / 1 | weak_mirnov_support | 5.88 (1279.07) | 85.09 (142.29) |
| cohort / 1 | screening_available | 99.87 (1282.13) | 100.00 (142.29) |
| cohort / 2 | rms_above_seed | 0.01 (1282.13) | 12.25 (142.29) |
| cohort / 2 | median5ms_above_seed | 0.00 (1282.13) | 12.12 (142.29) |
| cohort / 2 | rms_above_weak | 0.18 (1282.13) | 54.18 (142.29) |
| cohort / 2 | frequency_in_range_legacy | 15.51 (400.54) | 92.55 (142.29) |
| cohort / 2 | frequency_coherent50ms | 0.74 (400.54) | 46.66 (142.29) |
| cohort / 2 | mirnov_phase_fit | 1.18 (1279.07) | 27.93 (142.29) |
| cohort / 2 | mirnov_prominence | 71.86 (1279.07) | 99.51 (142.29) |
| cohort / 2 | span_coherent_support | 0.68 (1282.13) | 49.09 (142.29) |
| cohort / 2 | seed_coherent_support | 0.31 (1282.13) | 46.66 (142.29) |
| cohort / 2 | weak_mirnov_support | 2.38 (1279.07) | 95.93 (142.29) |
| cohort / 2 | screening_available | 99.87 (1282.13) | 100.00 (142.29) |
| population / 1 | rms_above_seed | 0.01 (14172.08) | 35.26 (1346.40) |
| population / 1 | median5ms_above_seed | 0.00 (14172.08) | 35.35 (1346.40) |
| population / 1 | rms_above_weak | 1.03 (14172.08) | 74.99 (1346.40) |
| population / 1 | frequency_in_range_legacy | 77.51 (3015.59) | 98.26 (1346.40) |
| population / 1 | frequency_coherent50ms | 0.50 (3015.59) | 60.57 (1346.40) |
| population / 1 | mirnov_phase_fit | 11.80 (14129.38) | 68.27 (1345.56) |
| population / 1 | mirnov_prominence | 72.28 (14129.38) | 99.24 (1345.56) |
| population / 1 | span_coherent_support | 4.61 (14172.08) | 71.24 (1346.40) |
| population / 1 | seed_coherent_support | 1.25 (14172.08) | 60.57 (1346.40) |
| population / 1 | weak_mirnov_support | 5.36 (14129.38) | 87.06 (1345.56) |
| population / 1 | screening_available | 99.80 (14172.08) | 100.00 (1346.40) |
| population / 2 | rms_above_seed | 0.01 (14172.08) | 13.15 (1346.40) |
| population / 2 | median5ms_above_seed | 0.00 (14172.08) | 12.90 (1346.40) |
| population / 2 | rms_above_weak | 0.18 (14172.08) | 56.04 (1346.40) |
| population / 2 | frequency_in_range_legacy | 16.31 (3015.59) | 93.57 (1346.40) |
| population / 2 | frequency_coherent50ms | 0.69 (3015.59) | 48.20 (1346.40) |
| population / 2 | mirnov_phase_fit | 0.90 (14129.38) | 28.19 (1345.56) |
| population / 2 | mirnov_prominence | 72.28 (14129.38) | 99.24 (1345.56) |
| population / 2 | span_coherent_support | 0.56 (14172.08) | 50.42 (1346.40) |
| population / 2 | seed_coherent_support | 0.31 (14172.08) | 48.20 (1346.40) |
| population / 2 | weak_mirnov_support | 1.97 (14129.38) | 96.54 (1345.56) |
| population / 2 | screening_available | 99.80 (14172.08) | 100.00 (1346.40) |

Source: [criterion_support_fix2.json](../../data/events/neoclassical_tearing_mode/benchmark/sources/criterion_support_fix2.json).

## Abrupt collapse and locking

An amplitude fall from above seed to below release within ≤5 ms is never `ended=decay`. It is `ended=locked` only when independently confirmed, otherwise `ended=unknown`. The post-collapse phase is uncertain until the lock signal falls or the discharge ends; without confirmation/release evidence it remains uncertain to the catalog discharge end. The fetched n1 `DUSBRADIAL` PTDATA radial-field detector is in **volts**, not gauss. Confirmation requires ≥5 V continuously for 20 ms within 100 ms of the collapse or eligible frequency drop; release requires the field to stay below 5 V for 200 ms, so shorter dips are bridged. This is a conservative local voltage convention, not a calibrated island-field measurement. `N1FREQ`/`N2FREQ` dropping to ≤1 kHz for 20 ms after rotation alone creates only a candidate. n2 has no independent radial confirmation.

| Set | Labeled shots | Intervals | Mode shots | Ends | Confirmed locks |
|---|---:|---:|---:|---|---:|
| cohort | 450 | 78 | 73 | {'decay': 48, 'locked': 18, 'plasma_end': 1, 'unknown': 11} | 18 |
| population | 4798 | 821 | 708 | {'decay': 448, 'locked': 226, 'plasma_end': 17, 'unknown': 130} | 226 |

Source: benchmark `label_counts`, `locking_coverage`, and the adjacent label metadata. Unknown status is not evidence that no lock occurred. Fetching runs only on the login node through `fdp run` and stops on the first auth error.

The `extend_tm_interval` table requires conversion before promotion to `review/`: TM catalog state 3 is forbidden, n-specific present/uncertain rows overlap, onset rows have zero length, and fractional-ms boundaries must become whole milliseconds. `test_shot_table_validates_extension_intervals_with_onset_points_and_spans` tests geometry, not catalog validity.

## Historical-onset agreement and limitations

Recall of the lab's archived onsets within 100 ms is **Seo 12/26** and **survival 16/67** on the development shots (13/26 and 18/67 on the earlier 500-shot labels). The strong 50 ms seed rule omits short and fast-locking modes: of the missed onsets, 9/14 (Seo) and 30/51 (survival) are short bursts, 4 and 16 lack a supported coherent line. Strict containment and miss reasons are in the linked JSONs. Survival agreement is near-circular because it shares RMS, 12 G, 50 ms and release conventions. The rule was not selected to maximize archive agreement.

| Reference | Covered shots | Onsets | Matched | Strict contained |
|---|---:|---:|---:|---:|
| seo | 80 | 26 | 12 | 9 |
| survival | 175 | 67 | 16 | 8 |
Sources: benchmark `agreement`, [agreement_seo_cohort_dev.json](../../data/events/neoclassical_tearing_mode/benchmark/sources/agreement_seo_cohort_dev.json), [agreement_survival_cohort_dev.json](../../data/events/neoclassical_tearing_mode/benchmark/sources/agreement_survival_cohort_dev.json). The frozen-rule sensitivity is generated by `tm_sensitivity.py` and bundled as `sensitivity_survival_dev.json`; it does not tune the rule.

## Development detection benchmark

The original seed-0 outer held-shot assignment is unchanged. Before any fit, `tm_cv_plan.py` freezes inner shot roles in `inner_splits_fix2.json`, using 10% per shot-has-an-interval stratum within each outer-training pool. Within-stratum swaps enforce observable positive support for every model and legacy target. No predicted score or held-fold performance chooses roles. Model availability is applied only after this shared plan. Each fit asserts positive-bearing early-stopping validation; threshold selection raises on zero positives and asserts no 0.999 fallback. Published CNN also appears at its own 0.5 threshold beside the retrained row. DSM's published survival threshold 0.7 corresponds to risk 0.3.

All learned models are refitted with three seeds. Prior hyperparameters remain fixed from earlier development work, so this is not fully nested hyperparameter selection. The scalar CNN has t+25 ms input lookahead and is an offline detector. DSM is a detection head on its original embedding; published DSM rows remain horizon forecasts, with known original training shots excluded from their reported held-out scores. Published CNN original training overlap is unknown (dagger), so those scores are descriptive.

Targets use absolute 10 ms bins. Legacy rows use their 25 ms native bins. Every row has its own input/reference-dependent shot set, given explicitly in its source JSON. Categories 2/3 and unavailable inputs/scores are excluded in primary rows. AUROC/AP use 1,024 score quantiles; all intervals use 1,000 whole-shot bootstrap draws (seed 0). Segmental F1 uses IoU 0.5, 50 ms minimum segments and ≤50 ms negative-gap closing; unavailable bins remain hard barriers and do not enter intersection or union.

| Model | Setting | Shots / bins | AUROC [95% CI] | AUPRC [95% CI] | F1 [95% CI] | Segmental F1 [95% CI] |
|---|---|---:|---|---|---|---|
 | tm-onsetcnn-published† | Legacy | 79 / 5580 | 0.897 [0.832,0.952] | 0.601 [0.385,0.799] | 0.462 [0.286,0.615] | — |
 | tm-dsm-500ms-published | Legacy | 80 / 8362 | 0.721 [0.614,0.828] | 0.294 [0.179,0.442] | 0.275 [0.167,0.364] | — |
 | tm-onsetcnn-published† | Interval | 265 / 25683 | 0.979 [0.963,0.991] | 0.938 [0.885,0.970] | 0.824 [0.727,0.888] | 0.672 [0.549,0.766] |
 | tm-dsm-500ms-published | Interval | 236 / 35868 | 0.776 [0.683,0.848] | 0.434 [0.298,0.572] | 0.452 [0.328,0.555] | 0.255 [0.130,0.381] |
 | tm-onsetcnn-retrained | Interval | 266 / 26005 | 0.957 [0.929,0.979] | 0.878 [0.762,0.950] | 0.796 [0.684,0.881] | 0.716 [0.568,0.820] |
 | tm-onsetcnn-published† | Interval, published thr. 0.5 | 265 / 25683 | 0.979 [0.963,0.991] | 0.938 [0.885,0.970] | 0.715 [0.594,0.813] | 0.604 [0.420,0.760] |
 | tm-dsm-retrained | Interval | 322 / 47875 | 0.946 [0.910,0.975] | 0.834 [0.726,0.912] | 0.673 [0.575,0.760] | 0.640 [0.512,0.729] |
 | tm-ours | Interval | 450 / 142103 | 0.999 [0.997,1.000] | 0.989 [0.976,0.997] | 0.952 [0.916,0.978] | 0.857 [0.775,0.928] |
 | tm-rms | Interval | 450 / 142103 | 0.987 [0.980,0.993] | 0.918 [0.868,0.955] | 0.763 [0.681,0.830] | 0.347 [0.273,0.439] |
 | tm-ours-rms | Interval | 450 / 142103 | 0.999 [0.997,1.000] | 0.986 [0.967,0.997] | 0.939 [0.900,0.968] | 0.843 [0.764,0.918] |
 | tm-ours | Uncertain = negative | 450 / 246801 | 0.963 [0.954,0.971] | 0.584 [0.471,0.685] | 0.425 [0.348,0.494] | 0.269 [0.215,0.321] |

The Mirnov-derived mask excludes about **42% of development catalog-window time**, equivalent to **49.6%** of current observable development plasma. It shares `tm-ours` inputs, so the task emphasizes strong modes versus quiet magnetic time and can favor the magnetic detector. The uncertain-as-negative sensitivity row uses identical fits and validation thresholds, scores category 2 as negative, and still excludes category 3. Finite uncertain features remain in temporal input context; the target mask controls loss/scoring only. The +RMS row adds circular label inputs.

### Paired comparison

Same 266 development shots / 26005 identical available 10 ms bins; each model retains its positive-bearing validation thresholds. Paired bootstrap draws resample the same shots. Ranking, not a threshold-specific F1 gain, is the primary comparison.

| Model | AUROC | AUPRC | F1 | Segmental F1 |
|---|---|---|---|---|
 | tm-ours | 0.998 [0.993,1.000] | 0.992 [0.974,1.000] | 0.950 [0.875,0.990] | 0.914 [0.840,0.970] |
 | tm-onsetcnn-retrained | 0.957 [0.929,0.979] | 0.878 [0.762,0.950] | 0.796 [0.684,0.881] | 0.716 [0.568,0.820] |
 | Difference, tm-ours−CNN | 0.041 [0.021,0.066] | 0.114 [0.049,0.213] | 0.154 [0.088,0.247] | 0.198 [0.080,0.344] |

Published/retrained CNN ranking on exactly 256 common shots and 22844 common 10 ms bins:

| Model | AUROC [95% CI] | AUPRC [95% CI] |
|---|---:|---:|
| tm-onsetcnn-published | 0.979 [0.964,0.990] | 0.941 [0.892,0.971] |
| tm-onsetcnn-retrained | 0.954 [0.926,0.977] | 0.880 [0.760,0.953] |

Source: benchmark `paired_common_shots` and `cnn_ranking_common_bins`. No retraining-generalization claim follows from published-CNN comparisons while original training overlap is unknown.

## Coverage and publication artifacts

**No TM coverage gain is claimed.** The preceding survival-matched like-for-like coverage was 435.7 s interval versus 909.9 s legacy; new conservative uncertainty further changes support. Current coverage uses the same measured plasma-start/catalog-end domain and 10 ms grid on each matched shot set. Observable time includes uncertainty; labeled time excludes it. Scoring can include quiet ramp-up, so scoring bins and observable-plasma coverage have different denominators.

| Set | Shots | Observable plasma s | Labeled s | Uncertain s |
|---|---:|---:|---:|---:|
| ours | 450 | 2109.09 | 1062.11 | 1046.98 |
| ours_population | 4798 | 22620.97 | 11748.25 | 10872.72 |

| Matched reference | Shots | Legacy labeled s | Interval labeled s | Common labeled s |
|---|---:|---:|---:|---:|
| legacy_seo | 80 | 129.98 | 171.50 | 47.09 |
| legacy_survival | 175 | 799.29 | 375.93 | 331.97 |

Sources: benchmark `coverage`, adjacent label metadata, and [figure2_tm.json](figure2_tm.json), which includes ranking/F1 intervals and exact like-for-like coverage. Cohort/population overlap and must not be added.

`tm_gallery.py --width 3.25 --columns 1` provides column-sized example panels with 7.5 pt text at final width. The 7.3-inch complete MHR/Mirnov galleries remain supplementary audit material and must not be shrunk into a paper column. Their JSON sidecars record shots, width, font, diagnostic, label/source hashes and image hashes. All changed PNGs are visually inspected. `tm_render_tables.py` renders the exact final TeX at 6.75-inch text width, rejects overfull horizontal boxes and writes PDF/150-dpi PNG previews with provenance. The appendix explains the dagger.

## Reproduction

Use the prescribed frozen/no-install pixi labelmaker environment, worktree PYTHONPATH, scratch TMPDIR and LABELER_NO_FETCH=1. Only the authorized DUSBRADIAL fetch unsets LABELER_NO_FETCH and runs through `fdp run` on the login node. Train on CUDA_VISIBLE_DEVICES=1 with the prescribed phase3 CUDA venv; never use `--final`.

Sequence: `tm_magfeatures.py`, `tm_label.py` (development and nonblind population), `tm_audit_rule.py`, `tm_cv_plan.py`, `tm_prior_retrain.py --model cnn/dsm`, `tm_ours.py --features magnetics/magnetics+rms` and `--baseline`, `tm_prior_published.py` (legacy and interval), `tm_agreement.py --exclude-test --tag _dev`, `tm_sensitivity.py`, `tm_gallery.py`, `tm_benchmark.py --rescore --gallery-reviewed`, `tm_write_doc.py`, `tm_render_tables.py`. Source JSONs and exact shot lists are committed under the benchmark's `sources/`; large predictions, weights, signals and figures stay under `$LABELER_ROOT/round4/tm/`.

Remaining research limitations: magnetic target/input sharing, local n2 and voltage conventions, omissions of weak/fast-locking modes, unknown published CNN overlap, incomplete weak-screen inputs, and no independent ECE island radius or fully nested hyperparameter selection.
