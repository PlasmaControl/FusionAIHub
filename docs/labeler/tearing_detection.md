# Whole-interval tearing-mode detection

The label is **a strong rotating n=1/n=2 mode (tearing-mode proxy)**. It describes
a magnetic signature, without independently identifying a tearing island or
separating classical from neoclassical modes. An observed onset is a point
(`iscrowd: 0`), and the whole rotating interval is a span (`iscrowd: 1`).
`tm-ours` reads the same Mirnov array as the label's N1RMS. This task measures
recovery of an RMS-based rule; the coherent-line requirement also uses magnetic
information shared with the detector. Removing the explicit RMS input does not
make this an independent physics validation.

The machine-readable result is
[`tm_benchmark.json`](../../data/events/neoclassical_tearing_mode/benchmark/tm_benchmark.json),
with the individual records in `benchmark/sources/`. Signals, features,
checkpoints, predictions, galleries and logs are external, under
`$LABELER_ROOT/round4/tm/`, with
`LABELER_ROOT=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker`.

## Frozen rule and calibration

`labeler.tearing.rule` uses `\\MHD::N1RMS` and `N2RMS` in gauss on their
approximately 1 kHz grid, after the Ip-derived plasma start within each catalog
window. Its strong-mode requirements are:

| Choice | Value | Source or status |
|---|---|---|
| n1 seed | raw and 5 ms median RMS >12 G continuously for ≥50 ms | Farre-Kaga duration/threshold; median confirmation is local |
| n2 seed | raw and median RMS >6 G continuously for ≥50 ms | local n2 extension |
| Rotating seed | n-resolved line 1.5–30 kHz, continuously supporting ≥50 ms of the seed | N1FREQ/N2FREQ, or Mirnov fallback |
| Frequency stability | 50 ms p90−p10 width ≤max(2 kHz, 25% of the median) | local rejection of rapid sweeps |
| Span support | ≥80% has an n-resolved 1–30 kHz line | local coherence requirement |
| Mirnov fallback | phase fit ≥0.9, prominence ≥10 dB, amplitude above development quiet p95 | local thresholds, frozen before final learning |
| Release | max(1 G, 10% of that continuous seed's peak) | local interval extension of peak-relative onset |
| Join release dips | ≤50 ms, only across available acquisition | local interval convention |
| Harmonic rejection | n2 RMS >0.57×n1 RMS | development-only calibration |
| Unobserved onset | start within 20 ms of the plasma/window opening | censoring convention |

[Farre-Kaga et al., *Interpreting AI for Fusion*](https://arxiv.org/abs/2502.20294)
requires n1 RMS above 12 G continuously for 50 ms, with onset at 10% of the
peak, H-mode constraints and flat-top selection. This implementation enforces
the continuous seed **before joining runs**: time in sub-seed or sub-release
gaps never supplies the hold. It still omits the H-mode restriction, adds n2,
and uses local line-stability and interval-boundary conventions. The
peak-relative boundary uses future samples and is an offline annotation.
Fu et al., *Physics of Plasmas* 27, 022501 (2020),
[doi:10.1063/1.5125581](https://doi.org/10.1063/1.5125581), used 10 G and 50 ms
and quiet negatives below 5 G. These sources do not independently validate the
local n2 seed or identify an island.

Each continuous seed expands backwards and forwards at its own release level.
Qualified components separated by short release dips can merge, retaining
`release_components` with their individual boundaries and thresholds. The
summary `release_g` is their minimum; `peak_g` is the largest qualified seed
peak. It is not a re-expansion of the complete merged span at 10% of its largest
peak. Acquisition gaps are preserved as NaN in resampling and smoothing and
cannot be joined. Every missing RMS stretch is category 3 (not observable).

Short seeds, broadband or unsupported bursts, and lines above 30 kHz become
category 2 (uncertain). Sustained coherent sub-seed RMS above the absent-time
p95 also becomes uncertain. A weak n-resolved Mirnov line can establish
uncertainty with a continuous ≥100 ms core above the development quiet-amplitude
p95, then extend along that line at 10% of the amplitude floor. Its available
interruptions ≤50 ms can join after the core is established. This uncertainty
hysteresis covers weak tracks such as 196494 and 187072 without declaring them
strong modes. Remaining measured catalog-window time is absent, including quiet
ramp-up; coverage tables separately clip to observable plasma. Rule firings in the
excluded ramp-up are uncertain. Categories 2 and 3 are excluded from training
and scoring.

`tm_calibrate_rule.py` derives the harmonic cutoff from strong n1-only modes on
**development shots only**: the p99 of n2/n1 is rounded upwards to 0.57. It also
records the absent-time RMS and coherent-amplitude distributions. The previous
0.4 cutoff cited blind test shot 187043. That reference was a test exposure and
has been removed; the replacement calibration does not open that shot's
signals. The calibration's reference shot lists and signal-inventory scope are
frozen in `calibration_dev_fix1.json`, and can be replayed after additional
population frequency fetching. Historical blind-test model output files also
exist from the earlier implementation; they are excluded from this benchmark.
No new blind-test training, threshold tuning or model scoring was performed.

## Locking and poloidal number

`N1FREQ`/`N2FREQ` dropping to ≤1 kHz for 20 ms after ≥1.5 kHz is a
`locked_candidate`. Every in-span drop, and drops within 100 ms after the RMS
end, is retained in `lock_candidates_ms`; `lock_time_ms` stores the earliest
candidate. A drop alone does not truncate the interval or set `locked=true`.
Only a matching independently confirmed locked-mode time truncates the rotating
span at that time. Regression tests exercise that confirmed branch. No dedicated
locked-mode diagnostic is resolved in the present data, so all reported locking
states are unknown (`locked_known=false`). For an interval without a usable
frequency record, `ended=unknown` is explicit. Complete unknown-shot lists are
in each label table's metadata, rather than a truncated example list.

Poloidal number requires `m=nq` at an independently observed island radius.
`labeler.tearing.surface.supported_m` can use EFIT q and such a radius, but this
pipeline resolves no ECE island radius. **m is not assigned**; a unique rational
surface in EFIT is insufficient radial evidence. Omnimode m is not used.

## Labels and visual audit

The cohort table is
[`extend_tm_interval/tm_interval.csv`](../../data/events/neoclassical_tearing_mode/extend_tm_interval/tm_interval.csv)
with adjacent metadata. Population labels and detailed interval tables are
external: `labels/tm_interval_population.csv`, its metadata,
`labels/tm_intervals_full_{cohort,population}.csv`, and
`labels/plasma_start_{cohort,population}.json`. Counts, original-to-final
changes, and an independent raw/median continuous-seed audit are bundled in
`label_counts` and `rule_audit`.

| Set | Requested / labeled | Mode shots | Intervals n1 / n2 | Onset points | Confirmed locks* | Candidates | Missing RMS |
|---|---:|---:|---:|---:|---:|---:|---:|
| Cohort | 500 / 500 | 85 | 75 / 19 | 94 | 0 | 37 | 0 |
| Population | 4,872 / 4,848 | 750 | 677 / 203 | 873 | 0 | 290 | 24 |

*Lock confirmation is unknown for every interval, including every population
interval. Zero confirmed locks is not a measured absence of locking. The final
strong intervals all have usable frequency records after the permitted fetches;
the dedicated confirmation diagnostic remains unavailable. Cohort and population
overlap and must not be added. Seven population intervals have censored onsets.
Sources: label metadata and
[`audit_fix1_current.json`](../../data/events/neoclassical_tearing_mode/benchmark/sources/audit_fix1_current.json).

Compared with the pre-fix tables, **228 of 273 cohort geometries changed**
(179 removed, 49 modified), and **1,900 of 2,323 population geometries changed**
(1,442 removed, 458 modified). Exact geometries retained are 45 and 423;
25 and 101 of those also changed old-field attributes. No new nonoverlapping
span was added. Counts refer to original spans, allowing a merged final span
to overlap more than one original span.

The independent audit finds **0/94 cohort and 0/880 population intervals**
without a continuous 50 ms raw or median seed crossing (0% in each set).
For n1 specifically, the Farre-Kaga duration deviation is 0/75 and 0/677;
the local n2 extension is 0/19 and 0/203. Previously the pooled raw failures were
129/273 (47.3%) and 1,057/2,323 (45.5%); median failures were 119/273 (43.6%)
and 943/2,323 (40.6%). The audit includes the final sample width in each hold
and never crosses an acquisition gap; its exact method and original snapshots
are identified in the same source record.

`tm_gallery.py --n 12 --seed 0` samples development shots, excluding blind test
shots. The MHR gallery uses live row 2 (rows 0, 1 and 7 are dead). MHR and Mirnov
panels show the 0–50 kHz spectrum, colored n1/n2 RMS, readable 6/12 G ticks,
matching legends, and gray uncertainty. Candidate hatch edges contrast with the
background; confirmed-lock hatching is distinct and would appear only for a
confirmed event. There are no confirmed locks in these galleries. Both random
and seven-shot physics PNGs were opened after the final rule freeze. Their
paths and source hashes are recorded in `gallery_review` and the rule audit.

| Reviewed shot | Final adjudication |
|---|---|
| 185953 | Former 3345–3679 ms span is uncertain: short seed; no present interval. |
| 194410 | Former 1126–1378 and 1930–2222 ms spans are uncertain; no present interval. |
| 186561 | Former n2 4011–4062 ms pulse is uncertain; no strong interval remains. |
| 190790 | Early rapid-sweep or quasistationary spans are uncertain; coherent n1 3163–5764 ms remains present, with unconfirmed locking candidates at 3163/5727 ms. |
| 195040 | Former n2 4213–5116 ms span at about 40 kHz is uncertain; no present interval. |
| 196494 | Weak n1 line is uncertain over 1270–5529 ms, including the visible 2–5 s track. |
| 187072 | Weak n2 line is uncertain over 1810–5180 ms. |

Source: the audit's `reviewed_shots` and
[`weak_diagnostics_fix1.json`](../../data/events/neoclassical_tearing_mode/benchmark/sources/weak_diagnostics_fix1.json).

## Agreement with historical onsets

`tm_agreement.py` reads survival onsets from `raw/tm_labels.h5`; Seo's 25 ms
growth-phase labels are aligned to physical time using `validate.archived_truth`.
They are not historical full present spans. Primary agreement requires a
same-n1 interval containing the onset or having an edge within 100 ms. Exact
zero-tolerance containment is reported separately. Error is reference onset
minus interval start, conditional on a match. Positive errors mean the interval
begins earlier. All-cohort comparisons audit labels; model selection and scoring
use only the 450 development shots.

**Survival agreement is near-circular**: the reference shares N1RMS, 12 G,
50 ms and a 10%-of-peak onset convention with this rule. It does not independently
validate tearing islands. The frozen-rule sensitivity was rerun using the actual
line masks, plasma starts and acquisition-gap policy; its variants characterize
sensitivity and did not choose the final rule.

| Reference / set | Covered shots | Onsets | Match / miss (100 ms) | Strict containment | Median error [Q25,Q75] ms | Mean absolute error ms | Comparable intervals without onset |
|---|---:|---:|---:|---:|---|---:|---:|
| Seo / development | 80 | 26 | 13 / 13 | 10 / 26 | 116 [73,221] | 203.3 | 3 / 15 |
| Seo / all cohort | 92 | 29 | 14 / 15 | 11 / 29 | 115.5 [77.25,216.25] | 195.2 | 3 / 16 |
| Survival / development | 175 | 67 | 18 / 49 | 10 / 67 | 2 [-10.25,136.25] | 202.0 | 5 / 23 |
| Survival / all cohort | 200 | 72 | 20 / 52 | 11 / 72 | 2 [-11.25,108.75] | 182.6 | 9 / 29 |

Sources: bundled `agreement_{seo,survival}_cohort_{dev,all}.json`, generated by
`tm_agreement.py`. Per-onset decisions are in the corresponding external CSVs.
Seo's 13 development misses are nine short seeds, three unsupported coherent
lines and one interval starting later. Survival's 49 misses are 30 short seeds,
14 unsupported lines, three below-seed events and two later intervals. Only
10 survival onsets lie literally inside a strong n1 span; eight more are admitted
by the 100 ms edge tolerance. The stricter seed/line requirements explain much
of the departure from the shared RMS reference. These mismatches were retained,
not shifted into agreement.

For development Seo, the three comparable intervals without an onset split into
one on a reference-mode shot and two on reference-quiet shots; 52 of all 67 n1
intervals fall outside usable reference coverage or have no reference. For
survival the corresponding five split into three and two, with 44 intervals
outside coverage. All-cohort splits are Seo one/two (59 outside) and survival
three/six (46 outside). Archives can supply one onset for a shot while the new
rule supplies multiple spans; lack of an onset also reflects this mismatch.
Reference-quiet spans are disagreements, not independent evidence of false
island detections.

| Frozen-rule survival sensitivity (development n1) | Intervals | Onset match fraction | Comparable intervals without onset |
|---|---:|---:|---:|
| Frozen: 12 G, 50 ms, 10% release | 67 | .269 | .217 |
| Seed 8 / 10 / 15 G | 99 / 86 / 49 | .478 / .403 / .149 | .220 / .206 / .286 |
| Hold 20 / 30 / 100 ms | 99 / 90 / 44 | .493 / .433 / .209 | .233 / .237 / .176 |
| Join gap 20 / 100 ms | 68 / 66 | .254 / .269 | .292 / .217 |
| Release 5% / 20% | 66 / 67 | .284 / .224 | .174 / .348 |

Source:
[`sensitivity_survival_dev.json`](../../data/events/neoclassical_tearing_mode/benchmark/sources/sensitivity_survival_dev.json),
rerun against the final label SHA and rule SHA. These are near-circular
sensitivity checks and are not competing selected label definitions.

## Detection evaluation

Targets are presence at centres of absolute 10 ms bins. Categories 2/3,
nonfinite inputs and nonfinite scores are unavailable. Ranking metrics use
pooled histograms at 1,024 score quantiles and approximate exact AUROC/AP.
Confidence intervals resample whole shots 1,000 times with seed 0; valid
bootstrap counts are in each source record.

Segmental F1 closes available negative gaps ≤50 ms and drops positive runs
shorter than 50 ms. **Unavailable bins are hard barriers**, and contribute
neither intersection nor union to temporal IoU. Greedy one-to-one matching uses
IoU 0.5 for the main table; 0.3/0.7 results are also saved. Legacy onset/growth
and horizon targets have no segmental F1.

All main rows, including the RMS baseline and unchanged published weights,
choose their thresholds inside the **same shot CV folds**. The full development
cohort is assigned five outer groups before filtering for each model's inputs;
seeded 10% inner validation shot lists are also shared. Each row selects its
own F1 threshold on its available inner-validation bins, then scores the held
outer shots. The source records list train/validation/held IDs, thresholds,
class counts and saved validation predictions. Fixed published thresholds occur
only in explicitly labelled extra rows. For DSM, upstream survival probability
0.7 means **risk 0.3**, because risk is one minus survival; the previous risk-0.7
provenance was incorrect. The main published DSM row uses the representative
500 ms horizon. Other horizons and fixed-threshold rows are in the appendix
export; they are forecasts compared to presence, not horizon-zero detectors.

Some shared inner validation subsets have no positive bins after the stricter
rule. In those folds an F1 optimum is unestimable, and the unchanged deterministic
highest-candidate fallback is flagged in the source records. Resulting threshold
F1 can be unstable; the independent threshold replay and paired ranking help
assess this limitation. The fold assignment was not changed to improve these
results.

`tm-onsetcnn` retrains the published scalar/profile CNN with a detection loss at t,
using three seeds. Its pinned scalar preprocessing reads some t+25 ms values,
so it remains an offline detector with that input lookahead. `tm-dsm` retrains
the published 38-feature embedding with a detection logit in place of the
survival mixture. Prior learning rates were selected on an earlier fold-0 inner
validation loss; they are fixed for this correction, so this is development CV,
without fully nested hyperparameter selection. Published DSM evaluations exclude
known original training shots. The CNN original training list is unavailable;
published CNN rows have unknown overlap and are descriptive.

`tm-ours` uses six Mirnov probes' spectral band-power and toroidal coherence
features, three dilated temporal convolutions and a three-seed ensemble. Its
main variant omits explicit RMS; the RMS-input variant is a circular ablation.
Feature normalization uses training shots only. In the corrected implementation,
inputs are zeroed only for nonfinite feature rows: the target-validity mask
controls loss/scoring, and no longer zeroes finite uncertain inputs. This removes
a target-derived dependency in the earlier model context. All learned variants
were retrained against the final frozen label table after this correction.
Saved fold ensembles, normalization and thresholds are in
`checkpoints/tm_ours_magnetics/` and `checkpoints/tm_ours_magnetics_rms/`.
The README calls them experimental CV checkpoints and does not designate
`tm-ours` as a deployed latest model.

| Model | Setting | Scored shots / bins | AUROC [95% CI] | AUPRC [95% CI] | F1 [95% CI] | Segmental F1 [95% CI] |
|---|---|---:|---|---|---|---|
| Published CNN | Legacy, overlap unknown | 79 / 5,580 | .897 [.832,.952] | .601 [.385,.799] | .567 [.372,.719] | — |
| Published DSM (500 ms) | Legacy, outside training | 80 / 8,362 | .721 [.614,.828] | .294 [.179,.442] | .157 [.089,.250] | — |
| Published CNN | Interval, overlap unknown | 262 / 25,527 | .982 [.969,.993] | .951 [.911,.978] | .539 [.375,.688] | .497 [.322,.667] |
| Published DSM (500 ms) | Interval, outside training | 235 / 35,426 | .796 [.711,.864] | .521 [.391,.657] | .454 [.345,.549] | .257 [.132,.413] |
| tm-onsetcnn | Tokamak-SI, retrained | 263 / 25,853 | .971 [.952,.986] | .925 [.859,.964] | .760 [.641,.854] | .718 [.570,.836] |
| tm-dsm | Tokamak-SI, retrained | 319 / 47,449 | .964 [.935,.986] | .906 [.842,.953] | .669 [.564,.753] | .540 [.420,.650] |
| tm-ours | Tokamak-SI, Mirnov | 450 / 141,884 | .998 [.996,.999] | .988 [.973,.996] | .872 [.815,.920] | .714 [.628,.800] |
| n1 RMS | Tokamak-SI, fold-tuned | 450 / 141,884 | .988 [.982,.994] | .928 [.882,.960] | .775 [.701,.833] | .370 [.291,.467] |
| tm-ours + RMS | Tokamak-SI, circular ablation | 450 / 141,884 | .998 [.997,1.000] | .987 [.972,.996] | .825 [.745,.892] | .708 [.615,.798] |

Source: benchmark `rows`, each linking its complete source JSON and exact
shot/bin/threshold definitions. The saved out-of-fold probabilities reproduce
all reported F1 and segmental F1 point values exactly. Histogram-ranking replay
differences from float32 serialization are below 1e-4. All saved learned
validation thresholds were independently replayed from their actual validation
probabilities. The published CNN can have better ranking and lower CV-threshold
F1 because its threshold calibration is weak in folds with no positives.
Input coverage differs across these rows; use the paired table for the direct
comparison of the learned detectors.

Both learned detectors are compared on **263 common shots / 25,853 identical
10 ms bins (258.53 s)**, preserving their original fold thresholds. Paired
bootstrap draws use the same sampled shots for both models.

| Model on common bins | AUROC [95% CI] | AUPRC [95% CI] | F1 [95% CI] | Segmental F1 [95% CI] |
|---|---|---|---|---|
| tm-ours | .996 [.988,1.000] | .985 [.959,.999] | .872 [.781,.949] | .873 [.782,.956] |
| tm-onsetcnn | .971 [.952,.986] | .925 [.858,.963] | .760 [.641,.854] | .718 [.570,.836] |
| Difference, ours−CNN | .024 [.012,.040] | .061 [.029,.115] | .112 [.015,.223] | .155 [.021,.304] |

Source: benchmark `paired_common_shots` and
[`table_tm_paired.tex`](table_tm_paired.tex). Its subset changes segment
availability, explaining why the magnetic detector's paired segmental F1 is
higher than its all-development segmental F1.

The historical pre-correction CNN retraining did **not** beat the published
weights on AUROC (.926 versus .930); it tied rounded AP (.778). Those historical
numbers used the old labels and belong only to the explicitly archived ranking
comparison. They are not the corrected rule's results. The final common-bin
published-versus-retrained comparison below states the current ranking.

**The retrained CNN does not beat the published weights on ranking.** On
253 common shots / 22,705 identical bins, published AUROC/AP are .982/.954;
retrained AUROC/AP are .970/.926. Retrained fold-threshold F1 is higher
(.758 versus .537), but that does not establish better ranking or generalization.
The published model's original training overlap is still unknown. Source:
benchmark `cnn_ranking_common_bins`.

## Paper exports and reproduction

[`table_tm_benchmark.tex`](table_tm_benchmark.tex) contains the main rows, with
a 78-word caption. The common-shot comparison is in
[`table_tm_paired.tex`](table_tm_paired.tex), and extra horizons/fixed-threshold
rows are in [`table_tm_benchmark_appendix.tex`](table_tm_benchmark_appendix.tex).
[`figure2_tm.json`](figure2_tm.json) supplies F1 **at fold-tuned thresholds** and
AUPRC with confidence bounds for legacy versus Tokamak-SI, plus like-for-like
coverage in observable plasma seconds. The matched-target/input comparisons
remain distinct from total label coverage; changing the label target is not a
paired performance improvement.

| Coverage on the same observable-plasma grid | Shots | Observable plasma s | Legacy labeled s | Interval labeled s on those shots | Common labeled s |
|---|---:|---:|---:|---:|---:|
| Seo-matched cohort | 92 | 451.03 | 147.19 | 199.01 | 52.93 |
| Survival-matched cohort | 200 | 984.76 | 909.91 | 435.68 | 383.83 |
| Whole cohort | 500 | 2,346.94 | — | 1,178.59 | — |
| Population supplement | 4,848 | 22,858.82 | — | 19,967.73 | — |

Source: benchmark `coverage` and Figure 2's `like_for_like_coverage`. Each
comparison uses the same physical 10 ms grid and plasma-start/window boundaries.
Observable seconds include uncertainty; labeled seconds exclude categories 2/3
and archive temporal holes. The old survival 1,200 s was the native fixed grid,
not observable plasma. Scoring-bin coverage in the results table can include
quiet catalog-window ramp-up; the plasma coverage table explicitly excludes it.
Much of the cohort now has uncertainty, and the population lacks Mirnov weak-line
features outside the cohort. Thus the population/cohort label coverage and weak
uncertainty coverage are different, rather than like-for-like performance cohorts.

Use the prescribed environment and temporary directory, `LABELER_NO_FETCH=1`
and CPU thread limits ≤8. Each long job uses `timeout 1800`, a log, and the
scratch `tmpsweep.sh` afterwards. `tm_label.py` regenerates cohort/population
labels; `tm_agreement.py` and `tm_sensitivity.py` regenerate onset checks;
`tm_prior_published.py`, `tm_prior_retrain.py`, and `tm_ours.py` regenerate CV
scores. The CUDA venv and GPU 1 were used for training within the 10 GB bound.
Never supply `--final`: that would score the blind split.

```bash
export TMPDIR=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/scratch/claude-89242e53/r4/tmp/tm
export LABELER_ROOT=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker
export LABELER_LABEL_TABLES=/scratch/gpfs/nc1514/FusionAIHub/data/events
export LABELER_NO_FETCH=1 PYTHONPATH=$PWD/src
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
 timeout 1800 pixi run --frozen --no-install --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e labelmaker python scripts/labeler/tm_benchmark.py --rescore --gallery-reviewed > "$LABELER_ROOT/round4/tm/logs/benchmark_reproduce.log" 2>&1
bash /scratch/gpfs/EKOLEMEN/nc1514/labelmaker/scratch/bin/tmpsweep.sh
```

`--gallery-reviewed` records an inspection already performed; open every PNG
before supplying it. Published CNN legacy/interval results have unknown training
overlap; historical test exposure, magnetic target/input sharing, locking
uncertainty and sparse positive validation folds remain limitations.
